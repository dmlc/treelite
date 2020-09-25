/*!
 * Copyright (c) 2020 by Contributors
 * \file xgboost_json.cc
 * \brief Frontend for xgboost model
 * \author Hyunsu Cho
 * \author William Hicks
 */

#include <algorithm>
#include <cstdio>
#include <cstdlib>
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
      sizeof(readBuffer));
  *out = std::move(ParseStream(std::move(input_stream)));
  fclose(fp);
}

void LoadXGBoostJSONModelString(const std::string &json_str, Model *out) {
  auto input_stream = std::make_unique<rapidjson::MemoryStream>(
      json_str.c_str(), json_str.size());
  *out = std::move(ParseStream(std::move(input_stream)));
}

}  // namespace frontend
}  // namespace treelite

namespace {

// TODO(wphicks): Use shared source for this
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
  explicit BaseHandler(std::weak_ptr<Delegator> parent_delegator) :
    delegator{parent_delegator} {};
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
  virtual bool EndObject(std::size_t memberCount) { return pop_handler(); }
  virtual bool StartArray() { return false; }
  virtual bool EndArray(std::size_t elementCount) { return pop_handler(); }

 protected:
  template <typename HandlerType, typename... ArgsTypes>
  bool push_handler(ArgsTypes &... args) {
    if (auto parent = delegator.lock()) {
      parent->push_delegate(std::make_shared<HandlerType>(delegator, args...));
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
    if (auto parent = delegator.lock()) {
      parent->pop_delegate();
      return true;
    } else {
      return false;
    }
  }

  void set_cur_key(const char *str, std::size_t length) {
    cur_key = std::string{str, length};
  }

  const std::string &get_cur_key() { return cur_key; }

  bool check_cur_key(const std::string &query_key) {
    return cur_key == query_key;
  }

  /* NOTE: Allowing non-const reference parameter because using pointer instead
   * *substantially* increases complexity of the implementation.
   */
  template <typename ValueType>
  bool assign_value(const std::string &key,
                    ValueType &output,  // NOLINT(runtime/references)
                    ValueType &&value) {
    if (check_cur_key(key)) {
      output = value;
      return true;
    } else {
      return false;
    }
  }

  template <typename ValueType>
  bool assign_value(const std::string &key,
                    ValueType &output,  // NOLINT(runtime/references)
                    const ValueType &value) {
    return assign_value(key, output, std::forward<ValueType>(value));
  }

 private:
  std::weak_ptr<Delegator> delegator;
  std::string cur_key;
};

class IgnoreHandler : public BaseHandler {
  using BaseHandler::BaseHandler;
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
  bool StartArray() { return push_handler<IgnoreHandler>(); }
};

template <typename OutputType> class OutputHandler : public BaseHandler {
 public:
  OutputHandler(std::weak_ptr<Delegator> parent_delegator,
                OutputType &output)  // NOLINT(runtime/references)
      : BaseHandler{parent_delegator}, m_output{output} {};
  OutputHandler(std::weak_ptr<Delegator> parent_delegator,
                OutputType &&output) = delete;  // NOLINT(runtime/references)

 protected:
  OutputType &m_output;
};

template <typename ElemType, typename HandlerType = BaseHandler>
class ArrayHandler : public OutputHandler<std::vector<ElemType>> {
 public:
  using OutputHandler<std::vector<ElemType>>::OutputHandler;
  bool Bool(ElemType b) {
    this->m_output.push_back(b);
    return true;
  }
  template <typename ArgType, typename IntType = ElemType>
  typename std::enable_if<std::is_integral<IntType>::value, bool>::type
  store_int(ArgType i) {
    this->m_output.push_back(static_cast<ElemType>(i));
    return true;
  }

  template <typename ArgType, typename IntType = ElemType>
  typename std::enable_if<!std::is_integral<IntType>::value, bool>::type
  store_int(ArgType i) {
    return false;
  }

  bool Int(int i) { return store_int<int>(i); }
  bool Uint(unsigned u) { return store_int<unsigned>(u); }
  bool Int64(int64_t i) { return store_int<int64_t>(i); }
  bool Uint64(uint64_t u) { return store_int<uint64_t>(u); }
  bool Double(ElemType d) {
    this->m_output.push_back(d);
    return true;
  }

  template <typename StringType = ElemType>
  typename std::enable_if<std::is_same<StringType, std::string>::value,
                          bool>::type
  store_string(const char *str, std::size_t length, bool copy) {
    this->m_output.push_back(ElemType{str, length});
    return true;
  }
  template <typename StringType = ElemType>
  typename std::enable_if<!std::is_same<StringType, std::string>::value,
                          bool>::type
  store_string(const char *str, std::size_t length, bool copy) {
    return false;
  }

  bool String(const char *str, std::size_t length, bool copy) {
    return store_string(str, length, copy);
  }

  bool StartObject(std::true_type) {
    this->m_output.emplace_back();
    return this->template push_handler<HandlerType, ElemType>(
        this->m_output.back());
  }

  bool StartObject(std::false_type) { return false; }

  bool StartObject() {
    return StartObject(
        std::integral_constant<bool, std::is_base_of<OutputHandler<ElemType>,
                                                     HandlerType>::value>{});
  }
};

class TreeParamHandler : public OutputHandler<int> {
 public:
  using OutputHandler<int>::OutputHandler;

  bool String(const char *str, std::size_t length, bool copy) {
    // Key "num_deleted" not documented in schema
    return (check_cur_key("num_feature") ||
            assign_value("num_nodes", m_output, atoi(str)) ||
            check_cur_key("size_leaf_vector") || check_cur_key("num_deleted"));
  }
};

class RegTreeHandler : public OutputHandler<treelite::Tree> {
 public:
  using OutputHandler<treelite::Tree>::OutputHandler;
  bool StartArray() {
    // Key "categories" not documented in schema
    // Key "split_type" not documented in schema
    return (
        push_key_handler<ArrayHandler<double>>("loss_changes", loss_changes) ||
        push_key_handler<ArrayHandler<double>>("sum_hessian", sum_hessian) ||
        push_key_handler<ArrayHandler<double>>("base_weights", base_weights) ||
        push_key_handler<IgnoreHandler>("categories") ||
        push_key_handler<ArrayHandler<int>>("leaf_child_counts",
                                            leaf_child_counts) ||
        push_key_handler<ArrayHandler<int>>("left_children", left_children) ||
        push_key_handler<ArrayHandler<int>>("right_children", right_children) ||
        push_key_handler<ArrayHandler<int>>("parents", parents) ||
        push_key_handler<ArrayHandler<int>>("split_indices", split_indices) ||
        push_key_handler<IgnoreHandler>("split_type") ||
        push_key_handler<ArrayHandler<double>>("split_conditions",
                                               split_conditions) ||
        push_key_handler<ArrayHandler<bool>>("default_left", default_left));
  }

  bool StartObject() {
    return push_key_handler<TreeParamHandler, int>("tree_param", num_nodes);
  }

  bool Uint(unsigned u) { return check_cur_key("id"); }

  bool EndObject(std::size_t memberCount) {
    m_output.Init();
    if (num_nodes != loss_changes.size() || num_nodes != sum_hessian.size() ||
        num_nodes != base_weights.size() ||
        num_nodes != leaf_child_counts.size() ||
        num_nodes != left_children.size() ||
        num_nodes != right_children.size() || num_nodes != parents.size() ||
        num_nodes != split_indices.size() ||
        num_nodes != split_conditions.size() ||
        num_nodes != default_left.size()) {
      return false;
    }

    std::queue<std::pair<int, int>> Q;  // (old ID, new ID) pair
    Q.push({0, 0});
    while (!Q.empty()) {
      int old_id, new_id;
      std::tie(old_id, new_id) = Q.front();
      Q.pop();

      if (left_children[old_id] == -1) {
        m_output.SetLeaf(new_id, base_weights[old_id]);
      } else {
        m_output.AddChilds(new_id);
        m_output.SetNumericalSplit(
            new_id, split_indices[old_id], split_conditions[old_id],
            default_left[old_id], treelite::Operator::kLT);
        m_output.SetGain(new_id, loss_changes[old_id]);
        Q.push({left_children[old_id], m_output.LeftChild(new_id)});
        Q.push({right_children[old_id], m_output.RightChild(new_id)});
      }
      m_output.SetSumHess(new_id, sum_hessian[old_id]);
    }
    return pop_handler();
  }

 private:
  std::vector<double> loss_changes;
  std::vector<double> sum_hessian;
  std::vector<double> base_weights;
  std::vector<int> leaf_child_counts;
  std::vector<int> left_children;
  std::vector<int> right_children;
  std::vector<int> parents;
  std::vector<int> split_indices;
  std::vector<double> split_conditions;
  std::vector<bool> default_left;
  int num_nodes;
};

class GBTreeModelHandler : public OutputHandler<treelite::Model> {
  using OutputHandler<treelite::Model>::OutputHandler;
  bool StartArray() {
    return (push_key_handler<ArrayHandler<treelite::Tree, RegTreeHandler>,
                             std::vector<treelite::Tree>>("trees",
                                                          m_output.trees) ||
            push_key_handler<IgnoreHandler>("tree_info"));
  }

  bool StartObject() {
    return push_key_handler<IgnoreHandler>("gbtree_model_param");
  }
};

class GradientBoosterHandler : public OutputHandler<treelite::Model> {
 public:
  using OutputHandler<treelite::Model>::OutputHandler;
  bool Uint(unsigned u) { return check_cur_key("num_boosting_round"); }
  bool String(const char *str, std::size_t length, bool copy) {
    fmt::string_view name{str, length};
    if (name != "gbtree") {
      LOG(ERROR) << "Only GBTree-type boosters are currently supported.";
      return false;
    } else {
      return true;
    }
  }

  bool StartObject() {
    if (push_key_handler<GBTreeModelHandler, treelite::Model>("model",
                                                              m_output)) {
      return true;
    } else {
      LOG(ERROR) << "Key \"" << get_cur_key()
                 << "\" not recognized. Is this a GBTree-type booster?";
      return false;
    }
  }
};

class ObjectiveHandler : public OutputHandler<std::string> {
  using OutputHandler<std::string>::OutputHandler;

  bool StartObject() {
    return (push_key_handler<IgnoreHandler>("reg_loss_param") ||
            push_key_handler<IgnoreHandler>("poisson_regression_param") ||
            push_key_handler<IgnoreHandler>("tweedie_regression_param") ||
            push_key_handler<IgnoreHandler>("softmax_multiclass_param") ||
            push_key_handler<IgnoreHandler>("lambda_rank_param") ||
            push_key_handler<IgnoreHandler>("aft_loss_param"));
  }

  bool String(const char *str, std::size_t length, bool copy) {
    return assign_value("name", m_output, std::string{str, length});
  }
};

class LearnerParamHandler : public OutputHandler<treelite::Model> {
 public:
  using OutputHandler<treelite::Model>::OutputHandler;
  bool String(const char *str, std::size_t length, bool copy) {
    return (assign_value("base_score", m_output.param.global_bias,
                         strtof(str, nullptr)) ||
            assign_value("num_class", m_output.num_output_group,
                         static_cast<int>(std::max(atoi(str), 1))) ||
            assign_value("num_feature", m_output.num_feature, atoi(str)));
  }
};

class LearnerHandler : public OutputHandler<treelite::Model> {
 public:
  using OutputHandler<treelite::Model>::OutputHandler;
  bool StartArray() { return push_key_handler<IgnoreHandler>("eval_metrics"); }

  bool StartObject() {
    // "attributes" key is not documented in schema
    return (push_key_handler<LearnerParamHandler, treelite::Model>(
                "learner_model_param", m_output) ||
            push_key_handler<GradientBoosterHandler, treelite::Model>(
                "gradient_booster", m_output) ||
            push_key_handler<ObjectiveHandler, std::string>("objective",
                                                            objective) ||
            push_key_handler<IgnoreHandler>("attributes"));
  }

  bool EndObject(std::size_t memberCount) {
    if (objective == "multi:softmax") {
      std::strncpy(m_output.param.pred_transform, "max_index",
                   sizeof(m_output.param.pred_transform));
    } else if (objective == "multi:softprob") {
      std::strncpy(m_output.param.pred_transform, "softmax",
                   sizeof(m_output.param.pred_transform));
    } else if (objective == "reg:logistic" || objective == "binary:logistic") {
      std::strncpy(m_output.param.pred_transform, "sigmoid",
                   sizeof(m_output.param.pred_transform));
    } else if (objective == "count:poisson" || objective == "reg:gamma" ||
               objective == "reg:tweedie") {
      std::strncpy(m_output.param.pred_transform, "exponential",
                   sizeof(m_output.param.pred_transform));
    } else {
      std::strncpy(m_output.param.pred_transform, "identity",
                   sizeof(m_output.param.pred_transform));
    }
    return pop_handler();
  }

 private:
  std::string objective;
};

class XGBoostModelHandler : public OutputHandler<treelite::Model> {
 public:
  using OutputHandler<treelite::Model>::OutputHandler;
  bool StartArray() {
    return push_key_handler<ArrayHandler<unsigned>, std::vector<unsigned>>(
        "version", version);
  }

  bool StartObject() {
    return push_key_handler<LearnerHandler, treelite::Model>("learner",
                                                             m_output);
  }

  bool EndObject(std::size_t memberCount) {
    if (memberCount != 2) {
      return false;
    }
    m_output.random_forest_flag = false;
    if (version[0] >= 1) {
      if (std::strcmp(m_output.param.pred_transform, "sigmoid") == 0) {
        m_output.param.global_bias =
            ProbToMargin::Sigmoid(m_output.param.global_bias);
      } else if (std::strcmp(m_output.param.pred_transform, "exponential") ==
                 0) {
        m_output.param.global_bias =
            ProbToMargin::Exponential(m_output.param.global_bias);
      }
    }
    return pop_handler();
  }

 private:
  std::vector<unsigned> version;
};

class RootHandler : public OutputHandler<treelite::Model> {
 public:
  using OutputHandler<treelite::Model>::OutputHandler;
  bool StartObject() {
    return push_handler<XGBoostModelHandler, treelite::Model>(m_output);
  }
};

class DelegatedHandler
    : public rapidjson::BaseReaderHandler<rapidjson::UTF8<>, DelegatedHandler>,
      public Delegator,
      public std::enable_shared_from_this<DelegatedHandler> {

 public:
  DelegatedHandler() : delegates{}, result{} {
    push_delegate(std::make_shared<RootHandler>(weak_from_this(), result));
  };

  void push_delegate(std::shared_ptr<BaseHandler> new_delegate) override {
    delegates.push(new_delegate);
  };
  void pop_delegate() override {
    delegates.pop();
  };
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
  DelegatedHandler handler{};
  rapidjson::Reader reader;

  rapidjson::ParseResult result = reader.Parse(*input_stream, handler);
  if (!result) {
    LOG(ERROR) << "Parsing error " << result.Code() << " at offset "
               << result.Offset();
    throw std::runtime_error(
        "Provided JSON could not be parsed as XGBoost model");
  }
  return handler.get_result();
}


}  // anonymous namespace
