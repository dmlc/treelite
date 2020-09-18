/*!
 * Copyright (c) 2020 by Contributors
 * \file xgboost_json.cc
 * \brief Frontend for xgboost model
 * \author Hyunsu Cho
 */

#include <cstdio>
#include <iostream>
#include <memory>
#include <queue>
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

// TODO: Use shared source for this
struct ProbToMargin {
  static float Sigmoid(float global_bias) {
    return -logf(1.0f / global_bias - 1.0f);
  }
  static float Exponential(float global_bias) { return logf(global_bias); }
};

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
  virtual bool EndArray(std::size_t elementCount) { return false; }

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

class IgnoreHandler : public BaseHandler {
  bool Null() { return true; }
  bool Bool(bool b) { return true; }
  bool Int(int i) { return true; }
  bool Uint(unsigned u) { return true; }
  bool Int64(int64_t i) { return true; }
  bool Uint64(uint64_t u) { return true; }
  bool Double(double d) { return true; }
  bool String(const char *str, std::size_t length, bool copy) { return true; }
  bool StartObject() { return push_handler<IgnoreHandler>(); }
  bool Key(const char *str, std::size_t length, bool copy) { return true; }
  bool EndObject(std::size_t memberCount) { return pop_handler(); }
  bool StartArray() { return push_handler<IgnoreHandler>(); }
  bool EndArray(std::size_t elementCount) { return pop_handler(); }
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
    this->m_output.emplace_back();
    return this->template push_handler<HandlerType, ElemType>(
        this->m_output.back());
  }

  bool EndArray(std::size_t elementCount) { return this->pop_handler(); }
};

class GBTreeModelParamHandler : public OutputHandler<treelite::Model> {
public:
  bool Uint(unsigned u) {
    return (assign_value("num_feature", m_output.num_feature,
                         static_cast<int>(u)) ||
            assign_value("num_output_group", m_output.num_output_group,
                         static_cast<int>(std::max(u, 1u))) ||
            check_cur_key("num_trees"));
  }
  bool EndObject(std::size_t memberCount) { return pop_handler(); }
};

struct Node {
  bool is_leaf;
  int child_left_id;
  int child_right_id;
  unsigned feature_id;
  treelite::tl_float threshold;
  treelite::tl_float leaf_output;
  bool default_left;
};

class TestNodeHandler : public OutputHandler<Node> {
public:
  TestNodeHandler(std::weak_ptr<Delegator> parent_delegator, Node &output)
      : OutputHandler{parent_delegator, output}, cur_index{0} {};

  bool Bool(bool b) {
    if (cur_index++ == 4) {
      m_output.default_left = b;
      return true;
    } else {
      return false;
    }
  }

  bool Double(double d) {
    if (cur_index++ == 3) {
      m_output.threshold = static_cast<treelite::tl_float>(d);
      return true;
    } else {
      return false;
    }
  }

  bool Uint(unsigned u) {
    switch (cur_index++) {
    case 0:
      m_output.child_left_id = static_cast<int>(u);
      return true;
    case 1:
      m_output.child_right_id = static_cast<int>(u);
      return true;
    case 2:
      m_output.feature_id = u;
      return true;
    default:
      return false;
    }
  }

  bool EndArray(std::size_t elementCount) {
    return (elementCount == 5 && pop_handler());
  }

private:
  int cur_index;
};

class NodeHandler : public OutputHandler<Node> {
  bool StartArray() {
    m_output.is_leaf = false;
    return push_handler<TestNodeHandler, Node>(m_output);
  }

  bool Double(double d) {
    m_output.is_leaf = true;
    m_output.leaf_output = static_cast<treelite::tl_float>(d);
    return true;
  }

  bool EndArray(std::size_t elementCount) { return pop_handler(); }
};

struct NodeStat {
  treelite::tl_float loss_chg;
  treelite::tl_float sum_hess;
  treelite::tl_float base_weight;
  int64_t instance_cnt;
};
class NodeStatHandler : public OutputHandler<NodeStat> {
public:
  NodeStatHandler(std::weak_ptr<Delegator> parent_delegator, NodeStat &output)
      : OutputHandler{parent_delegator, output}, cur_index{0} {};

  bool Double(double d) {
    switch (cur_index++) {
    case 0:
      m_output.loss_chg = static_cast<treelite::tl_float>(d);
      return true;
    case 1:
      m_output.sum_hess = static_cast<treelite::tl_float>(d);
      return true;
    case 2:
      m_output.base_weight = static_cast<treelite::tl_float>(d);
      return true;
    default:
      return false;
    }
  }

  bool Uint(unsigned u) {
    if (cur_index++ == 4) {
      m_output.instance_cnt = static_cast<int64_t>(u);
      return true;
    } else {
      return false;
    }
  }

  bool Uint64(unsigned u) {
    if (cur_index++ == 4) {
      m_output.instance_cnt = static_cast<int64_t>(u);
      return true;
    } else {
      return false;
    }
  }

  bool EndArray(std::size_t elementCount) { return (elementCount == 4); }

private:
  int cur_index;
};

class RegTreeHandler : public OutputHandler<treelite::Tree> {
public:
  bool StartArray() {
    return (push_key_handler<ArrayHandler<Node, NodeHandler>>("nodes", nodes) ||
            push_key_handler<ArrayHandler<NodeStat, NodeStatHandler>>("stats",
                                                                      nodes));
  }

  bool EndObject() {
    if (nodes.size() != stats.size()) {
      return false;
    }
    std::queue<std::pair<int, int>> Q; // (old ID, new ID) pair
    Q.push({0, 0});
    while (!Q.empty()) {
      int old_id, new_id;
      std::tie(old_id, new_id) = Q.front();
      Q.pop();
      if (old_id >= nodes.size()) {
        return false;
      }
      const Node &node = nodes[old_id];
      const NodeStat &stat = stats[old_id];

      if (node.is_leaf) {
        m_output.SetLeaf(new_id, node.leaf_output);
      } else {
        m_output.AddChilds(new_id);
        m_output.SetNumericalSplit(new_id, node.feature_id, node.threshold,
                                   node.default_left, treelite::Operator::kLT);
        m_output.SetGain(new_id, stat.loss_chg);
        Q.push({node.child_left_id, m_output.LeftChild(new_id)});
        Q.push({node.child_right_id, m_output.RightChild(new_id)});
      }
      m_output.SetSumHess(new_id, stat.sum_hess);
    }
    return true;
  }

private:
  std::vector<Node> nodes;
  std::vector<NodeStat> stats;
};

class GBTreeModelHandler : public OutputHandler<treelite::Model> {
  bool StartArray() {
    return (push_key_handler<ArrayHandler<treelite::Tree, RegTreeHandler>,
                             std::vector<treelite::Tree>>("trees",
                                                          m_output.trees) ||
            push_key_handler<IgnoreHandler>("tree_info"));
  }

  bool StartObject() {
    return push_key_handler<GBTreeModelParamHandler, treelite::Model>(
        "model_param", m_output);
  }

  bool EndObject(std::size_t memberCount) { return pop_handler(); }
};

class GradientBoosterHandler : public OutputHandler<treelite::Model> {
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
    if (push_key_handler<IgnoreHandler>("gbtree_train_param") ||
        push_key_handler<IgnoreHandler>("updater_train_param") ||
        push_key_handler<GBTreeModelHandler, treelite::Model>("model",
                                                              m_output)) {
      return true;
    } else {
      LOG(ERROR) << "Key \""
                 << std::string{get_cur_key().data(), get_cur_key().size()}
                 << "\" not recognized. Is this a GBTree-type booster?";
      return false;
    }
  }
  bool EndObject(std::size_t memberCount) { return pop_handler(); }
  bool StartArray() { return push_key_handler<IgnoreHandler>("updaters"); }
};

struct Objective {
  fmt::string_view name;
  unsigned num_class;
  bool output_prob;
  fmt::string_view loss_type;
  double scale_pos_weight;
  unsigned num_pairsample;
  double fix_list_weight;
  double max_delta_step;
  double tweedie_variance_power;
};

class ObjectiveHandler : public OutputHandler<Objective> {
  bool String(const char *str, std::size_t length, bool copy) {
    fmt::string_view value{str, length};
    return (assign_value("name", value, m_output.name) ||
            assign_value("loss_type", value, m_output.loss_type));
  }

  bool Uint(unsigned u) {
    return (assign_value("num_class", u, m_output.num_class) ||
            assign_value("num_pairsample", u, m_output.num_pairsample));
  }

  bool Double(double d) {
    treelite::tl_float value = static_cast<treelite::tl_float>(d);
    return (assign_value("scale_pos_weight", d, m_output.scale_pos_weight) ||
            assign_value("fix_list_weight", d, m_output.fix_list_weight) ||
            assign_value("max_delta_step", d, m_output.max_delta_step) ||
            assign_value("tweedie_variance_power", d,
                         m_output.tweedie_variance_power));
  }

  bool EndObject(std::size_t memberCount) { return pop_handler(); }
};

class LearnerHandler : public OutputHandler<treelite::Model> {
public:
  bool StartArray() { return push_key_handler<IgnoreHandler>("eval_metrics"); }

  bool StartObject() {
    return (
        push_key_handler<IgnoreHandler>("learner_train_param") ||
        push_key_handler<GradientBoosterHandler, treelite::Model>(
            "gradient_booster", m_output) ||
        push_key_handler<ObjectiveHandler, Objective>("objective", objective));
  }

  bool Double(double d) {
    return assign_value("base_score", m_output.param.global_bias,
                        static_cast<treelite::tl_float>(d));
  }
  bool Uint(unsigned u) {
    return check_cur_key("num_feature") || check_cur_key("num_class");
  }

  bool EndObject(std::size_t memberCount) {
    if (objective.name.compare("SoftMultiClassObj")) {
      if (objective.output_prob) {
        std::strncpy(m_output.param.pred_transform, "softmax",
                     sizeof(m_output.param.pred_transform));
      } else {
        std::strncpy(m_output.param.pred_transform, "max_index",
                     sizeof(m_output.param.pred_transform));
      }
    } else if (objective.name.compare("RegLossObj") &&
               (objective.loss_type.compare("LogisticRegression") ||
                objective.loss_type.compare("LogisticClassification"))) {
      std::strncpy(m_output.param.pred_transform, "sigmoid",
                   sizeof(m_output.param.pred_transform));
      m_output.param.sigmoid_alpha = 1.0f;
    } else if (objective.name.compare("GammaRegression") ||
               objective.name.compare("TweedieRegression") ||
               objective.name.compare("PoissonRegression")) {
      std::strncpy(m_output.param.pred_transform, "exponential",
                   sizeof(m_output.param.pred_transform));
    } else {
      std::strncpy(m_output.param.pred_transform, "identity",
                   sizeof(m_output.param.pred_transform));
    }
    return pop_handler();
  }

private:
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
    m_output.random_forest_flag = false;
    if (version.at(0) >= 1) {
      if (std::strcmp(m_output.param.pred_transform, "logistic")) {
        m_output.param.global_bias =
            ProbToMargin::Sigmoid(m_output.param.global_bias);
      } else if (std::strcmp(m_output.param.pred_transform, "exponential")) {
        m_output.param.global_bias =
            ProbToMargin::Exponential(m_output.param.global_bias);
      }
    }
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
